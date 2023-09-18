package hello;

public class TimesAdd {
	public static void main(String[] args) {
		int num = 1;
		int total = 0;
		do {
			int num2 = 1;
			int times = 1;
			while(num2<=num) {
				times*=num2;
				num2 ++;
			}
			total+=times;
			num++;
		}while(num<=10);
		System.out.println(total);
	}
}
