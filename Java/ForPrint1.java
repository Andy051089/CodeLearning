package hello;

public class ForPrint1 {
	public static void main(String[] args) {
		int num = 1;
		while(num<200) {
			if(num%7==0 && num%4!=0) {
			System.out.println(num);
			}
		num++;
		}
	}
}
