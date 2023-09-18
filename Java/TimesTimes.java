package hello;
public class TimesTimes {
	public static void main(String[] args) {
		int total = 1;
		int num = 1;
		while(num<=10) {
			total = total*num;
			num++;
		}
		System.out.print(total);
	}
}
